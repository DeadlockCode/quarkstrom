use std::{
    f32::consts::TAU,
    time::{Duration, Instant}, num::NonZeroU32,
};

use bytemuck::{Pod, Zeroable};

use gui::Gui;
use rand::{Rng, rngs::ThreadRng};
use winit::{
    dpi::PhysicalSize,
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use wgpu::util::DeviceExt;
use winit_input_helper::WinitInputHelper;

mod gui;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Camera {
    position: [f32; 2],
    scale: f32,
    aspect: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct World {
    size: f32,
    num_particles: u32,
    num_colors: u32,
    universal_repulsive_strength: f32,
    repulsive_distance: f32,
    interaction_distance: f32,
    interaction_multiplier: f32,
    velocity_half_life: f32,
    dt: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
    color: u32,
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

const VERTICES: [Vertex; 3] = [
    Vertex {
        position: [0.0, 2.0],
    },
    Vertex {
        position: [-1.73205080757, -1.0],
    },
    Vertex {
        position: [1.73205080757, -1.0],
    },
];

impl Particle {
    const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![1 => Float32x2, 2 => Float32x2, 3 => Uint32];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Particle>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
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
    input: WinitInputHelper,
    compute_pipeline: wgpu::ComputePipeline,
    particles: Vec<Particle>,
    particle_buffer: wgpu::Buffer,
    particle_bind_group: wgpu::BindGroup,
    attraction_matrix: Vec<f32>,
    attraction_matrix_buffer: wgpu::Buffer,
    attraction_matrix_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    world: World,
    world_buffer: wgpu::Buffer,
    world_bind_group: wgpu::BindGroup,

    gui: Gui,
    rng: ThreadRng,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        let world_size = 500.0;
        let num_particles = 64 * 100;
        let num_colors = 7;

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
                    features: wgpu::Features::empty(),
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
            .filter(|f| f.describe().srgb)
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

        let gui = Gui::new(&window, config.format, &device, &queue);

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let camera = Camera {
            position: [0.; 2],
            scale: world_size,
            aspect: config.width as f32 / config.height as f32,
        };

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
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

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let world = World {
            size: world_size,
            num_particles,
            num_colors,
            universal_repulsive_strength: 50.0,
            repulsive_distance: 5.0,
            interaction_distance: 100.0,
            interaction_multiplier: 5.0,
            velocity_half_life: 0.02,
            dt: 0.01,
        };

        let world_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("World Buffer"),
            contents: bytemuck::cast_slice(&[world]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let world_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("World Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let world_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("World Bind Group"),
            layout: &world_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: world_buffer.as_entire_binding(),
            }],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&world_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), Particle::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
                conservative: false,
            },
            depth_stencil: None, // 1.
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let particles = (0..64*256)
            .map(|_| {
                Particle {
                    position: [0.; 2],
                    velocity: [0.; 2],
                    color: 0,
                }
            })
            .collect::<Vec<_>>();
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let particle_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Particle Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Bind Group"),
            layout: &particle_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buffer.as_entire_binding(),
            }],
        });
        let attraction_matrix = (0..256 * 256)
            .map(move |_| 0.0)
            .collect::<Vec<_>>();
        let attraction_matrix_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Attraction Matrix Buffer"),
                contents: bytemuck::cast_slice(&attraction_matrix),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let attraction_matrix_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Attraction Matrix Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let attraction_matrix_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Attraction Matrix Bind Group"),
            layout: &attraction_matrix_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: attraction_matrix_buffer.as_entire_binding(),
            }],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &world_bind_group_layout,
                    &particle_bind_group_layout,
                    &attraction_matrix_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: "cs_main",
        });

        let input = WinitInputHelper::new();

        let mut res = Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            input,
            compute_pipeline,
            particles,
            particle_buffer,
            particle_bind_group,
            attraction_matrix,
            attraction_matrix_buffer,
            attraction_matrix_bind_group,
            render_pipeline,
            vertex_buffer,
            camera,
            camera_buffer,
            camera_bind_group,
            world,
            world_buffer,
            world_bind_group,
            gui,
            rng: rand::thread_rng(),
        };
        res.world.size = res.gui.state.world_size;
        res.world.num_particles = res.gui.state.num_particles;
        res.world.num_colors = res.gui.state.num_colors;
        res.attraction_matrix = if res.gui.state.random_attraction {
            (0..res.world.num_colors * res.world.num_colors)
                .map(|_| res.rng.gen_range(-1.0..1.0))
                .collect::<Vec<_>>()
        } else {
            res.gui.state.attraction_matrix.clone()
        };
        res.restart();
        res
    }

    pub fn restart(&mut self) {
        self.queue
            .write_buffer(&self.world_buffer, 0, bytemuck::cast_slice(&[self.world]));

        self.queue
            .write_buffer(&self.attraction_matrix_buffer, 0, bytemuck::cast_slice(&self.attraction_matrix));

        self.particles = (0..self.world.num_particles)
            .map(|_| {
                let r = self.rng.gen_range(0f32..1.0);
                let a = self.rng.gen_range(0f32..TAU);
                let d = 100.0;

                let color = self.rng.gen_range(0..self.world.num_colors);

                Particle {
                    position: [
                        d * r.sqrt() * a.cos() + self.world.size * 0.5,
                        d * r.sqrt() * a.sin() + self.world.size * 0.5,
                    ],
                    velocity: [0.; 2],
                    color,
                }
            })
            .collect::<Vec<_>>();
        self.queue
            .write_buffer(&self.particle_buffer, 0, bytemuck::cast_slice(&self.particles));

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

            self.camera.aspect = new_size.width as f32 / new_size.height as f32;
        }
    }

    fn input(&mut self, event: &Event<()>) {
        if self.input.update(event) {
            // Zoom relative to mouse position
            if let Some((mx, my)) = self.input.mouse() {
                let scroll_diff = self.input.scroll_diff() * 0.1;

                let x = mx / self.size.width as f32 * 2.0 - 1.0;
                let y = my / self.size.width as f32 * 2.0 - self.camera.aspect.recip();

                self.camera.position[0] += x * self.camera.scale * scroll_diff;
                self.camera.position[1] -= y * self.camera.scale * scroll_diff;

                self.camera.scale *= 1.0 - scroll_diff;
            }

            // Move camera
            let (mdx, mdy) = self.input.mouse_diff();
            if self.input.mouse_held(2) {
                self.camera.position[0] -= mdx / self.size.width as f32 * self.camera.scale * 2.0;
                self.camera.position[1] += mdy / self.size.width as f32 * self.camera.scale * 2.0;
            }
        }
        self.gui.input(&self.window, event);
    }

    fn update(&mut self) {
        self.gui.update(&self.window);


        if self.gui.state.restart {
            self.world.size = self.gui.state.world_size;
            self.world.num_particles = self.gui.state.num_particles;
            self.world.num_colors = self.gui.state.num_colors;
            self.attraction_matrix = if self.gui.state.random_attraction {
                (0..self.world.num_colors * self.world.num_colors)
                    .map(|_| self.rng.gen_range(-1.0..1.0))
                    .collect::<Vec<_>>()
            } else {
                self.gui.state.attraction_matrix.clone()
            };
            self.restart();
        }

        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
            });

            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.world_bind_group, &[]);
            compute_pass.set_bind_group(1, &self.particle_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.attraction_matrix_bind_group, &[]);
            compute_pass.dispatch_workgroups((self.world.num_particles as f32 / 64.0).ceil() as u32, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.world_bind_group, &[]);
            render_pass.set_bind_group(1, &self.camera_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_vertex_buffer(1, self.particle_buffer.slice(..));
            render_pass.draw(0..VERTICES.len() as u32, 0..self.particles.len() as u32);

            self.gui.render(&self.device, &self.queue, &mut render_pass);
        }

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

pub async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Quarkstrom")
        .with_inner_size(PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(window).await;

    let mut frames = 0;
    let mut start_time = Instant::now();

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
                state.update();
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
                frames += 1;
                if start_time.elapsed() > Duration::from_secs(1) {
                    println!("{}", frames);
                    start_time = Instant::now();
                    frames = 0;
                }

                // RedrawRequested will only trigger once, unless we manually
                // request it.
                state.window().request_redraw();
            }
            _ => {}
        }
    });
}
