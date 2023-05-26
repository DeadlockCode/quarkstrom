pub trait Gui : Send + Clone {
    fn new() -> Self;
    fn update(&mut self, ui: &mut imgui::Ui);
}

pub struct GuiHandler<GUI: Gui> {
    imgui: imgui::Context,
    renderer: imgui_wgpu::Renderer,
    platform: imgui_winit_support::WinitPlatform,

    pub gui: GUI,
}

impl<GUI: Gui> GuiHandler<GUI> {
    pub fn new(
        window: &winit::window::Window,
        format: wgpu::TextureFormat,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self
    where
        GUI: Gui,
    {
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

            gui: GUI::new(),
        }
    }

    pub fn input(&mut self, window: &winit::window::Window, event: &winit::event::Event<()>) {
        self.platform
            .handle_event(self.imgui.io_mut(), window, event);
    }

    pub fn update(&mut self, window: &winit::window::Window) {
        self.platform
            .prepare_frame(self.imgui.io_mut(), &window)
            .expect("Failed to prepare gui frame");
        let ui = self.imgui.frame();

        self.gui.update(ui);

        self.platform.prepare_render(ui, window);
    }

    pub fn render<'a>(
        &'a mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        self.renderer
            .render(self.imgui.render(), queue, device, render_pass)
            .expect("Rendering gui failed");
    }
}
