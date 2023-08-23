use egui::{ClippedPrimitive, Context};
use egui_wgpu::renderer::ScreenDescriptor;

pub struct GuiHandler {
    ctx: egui::Context,
    pub renderer: egui_wgpu::Renderer,
    state: egui_winit::State,
}

impl GuiHandler {
    pub fn new(
        window: &winit::window::Window,
        format: wgpu::TextureFormat,
        device: &wgpu::Device,
    ) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(&window);

        let renderer = egui_wgpu::Renderer::new(device, format, None, 1);

        Self {
            ctx,
            renderer,
            state,
        }
    }

    pub fn handle_event(&mut self, event: &winit::event::Event<()>) -> bool {
        match event {
            winit::event::Event::WindowEvent {
                window_id: _,
                event,
            } => self.state.on_event(&self.ctx, event).consumed,
            _ => false,
        }
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        window: &winit::window::Window,
        encoder: &mut wgpu::CommandEncoder,
        gui: &mut dyn FnMut(&Context),
    ) -> (Vec<ClippedPrimitive>, ScreenDescriptor) {
        let screen_descriptor = {
            let size = window.inner_size();
            egui_wgpu::renderer::ScreenDescriptor {
                size_in_pixels: [size.width, size.height],
                pixels_per_point: window.scale_factor() as f32,
            }
        };

        let raw_input: egui::RawInput = self.state.take_egui_input(window);
        self.ctx.begin_frame(raw_input);
        gui(&self.ctx);
        let full_output = self.ctx.end_frame();

        self.state
            .handle_platform_output(window, &self.ctx, full_output.platform_output);

        let clipped_primitives = self.ctx.tessellate(full_output.shapes);

        self.renderer.update_buffers(
            device,
            queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );
        for (tex_id, img_delta) in full_output.textures_delta.set {
            self.renderer
                .update_texture(device, queue, tex_id, &img_delta);
        }
        for tex_id in full_output.textures_delta.free {
            self.renderer.free_texture(&tex_id);
        }

        (clipped_primitives, screen_descriptor)
    }
}
