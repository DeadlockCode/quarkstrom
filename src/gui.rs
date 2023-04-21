pub struct State {
    pub restart: bool,
    pub world_size: f32,
    pub num_particles: u32,
    pub num_colors: u32,
    pub random_attraction: bool,
    pub attraction_matrix: Vec<f32>,
}

impl Default for State {
    fn default() -> Self {
        Self { 
            restart: false,
            world_size: 500.0,
            num_particles: 2000,
            num_colors: 5,
            random_attraction: true,
            attraction_matrix: vec![0.0],
        }
    }
}

pub struct Gui {
    imgui: imgui::Context,
    renderer: imgui_wgpu::Renderer,
    platform: imgui_winit_support::WinitPlatform,

    pub state: State,
}

impl Gui {
    pub fn new(
        window: &winit::window::Window,
        format: wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut imgui = imgui::Context::create();
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        platform.attach_window(
            imgui.io_mut(),
            &window,
            imgui_winit_support::HiDpiMode::Default,
        );
        imgui.set_ini_filename(None);

        let hidpi_factor = window.scale_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        imgui
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    oversample_h: 1,
                    pixel_snap_h: true,
                    size_pixels: font_size,
                    ..Default::default()
                }),
            }]);

        let renderer_config = imgui_wgpu::RendererConfig {
            texture_format: format,
            ..Default::default()
        };

        let renderer = imgui_wgpu::Renderer::new(&mut imgui, device, queue, renderer_config);

        Self { 
            imgui,
            renderer,
            platform,

            state: Default::default(),
        }
    }

    pub fn input(&mut self, window: &winit::window::Window, event: &winit::event::Event<()>) {
        self.platform
            .handle_event(self.imgui.io_mut(), window, event);
    }

    pub fn update(&mut self, window: &winit::window::Window) {
        fn hue2rgb(c: f32) -> [f32; 4] {
            return [
                (((c + 1.0).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0),
                (((c + 2.0 / 3.0).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0),
                (((c + 1.0 / 3.0).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0),
                1.0,
            ];
        }

        self.platform
            .prepare_frame(self.imgui.io_mut(), &window)
            .expect("Failed to prepare frame");
        let ui = self.imgui.frame();

        {
            ui.window("Quarkstrom")
                .size([300.0, 600.0], imgui::Condition::FirstUseEver)
                .build(|| {
                    self.state.restart = ui.button("Restart");
                    ui.separator();
                    ui.slider("World size", 0.0, 1000.0, &mut self.state.world_size);
                    ui.slider("Particles", 0, 64 * 256, &mut self.state.num_particles);
                    if ui.slider("Colors", 1, 256, &mut self.state.num_colors) {
                        self.state.attraction_matrix = vec![0f32; self.state.num_colors as usize * self.state.num_colors as usize];
                    }

                    ui.checkbox("Random Attraction Matrix", &mut self.state.random_attraction);
                    if !self.state.random_attraction {
                        ui.color_button("WOWO", [0.; 4]);
                        for j in 0..self.state.num_colors as usize {
                            let _num_id = ui.push_id_usize(j);
                            ui.same_line();
                            ui.color_button("WOWO", hue2rgb(j as f32 / self.state.num_colors as f32));
                        }
                        for i in 0..self.state.num_colors as usize {
                            let _label_id = ui.push_id_usize(i);
                            ui.color_button("WOWO", hue2rgb(i as f32 / self.state.num_colors as f32));
                            for j in 0..self.state.num_colors as usize {
                                let _num_id = ui.push_id_usize(j);
                                ui.same_line();
                                let attraction = self.state.attraction_matrix[i + j * self.state.num_colors as usize];
                                let r = if attraction.is_sign_negative() {-attraction } else { 0.0 };
                                let g = if attraction.is_sign_positive() { attraction } else { 0.0 };
                                if ui.color_button("WOWO", [r, g, 0., 0.]) {
                                    self.state.attraction_matrix[i + j * self.state.num_colors as usize] += 0.1;
                                    if self.state.attraction_matrix[i + j * self.state.num_colors as usize] > 1.01 {
                                        self.state.attraction_matrix[i + j * self.state.num_colors as usize] = -1.0;
                                    }
                                }
                            }
                        }
                    }
                    
                    let mouse_pos = ui.io().mouse_pos;
                    ui.text(format!(
                        "Mouse Position: ({:.1},{:.1})",
                        mouse_pos[0], mouse_pos[1]
                    ));
                });
        }
        self.platform.prepare_render(ui, window);
    }

    pub fn render<'a>(
        &'a mut self, 
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        self.renderer
            .render(
                self.imgui.render(),
                queue,
                device,
                render_pass,
            )
            .expect("Rendering failed");
    }
}
