use std::{thread, time::Duration};

use quarkstrom::xyz;
use ultraviolet::Vec3;
use winit::dpi::PhysicalSize;

fn main() {
    let config = xyz::Config {
        window_size: PhysicalSize::new(1600, 900),
    };
    xyz::run::<Renderer>(config);

    loop {
        thread::sleep(Duration::from_secs(1));
    }
}

const CUBE: [Vec3; 8] = [
    Vec3::new(-0.5, 0.5, 0.5),
    Vec3::new(0.5, 0.5, 0.5),
    Vec3::new(0.5, 0.5, -0.5),
    Vec3::new(-0.5, 0.5, -0.5),
    Vec3::new(-0.5, -0.5, 0.5),
    Vec3::new(0.5, -0.5, 0.5),
    Vec3::new(0.5, -0.5, -0.5),
    Vec3::new(-0.5, -0.5, -0.5),
];

struct Renderer {

}

impl xyz::Renderer for Renderer {
    fn new() -> Self {
        Self {

        }
    }

    fn input(&mut self, input: &winit_input_helper::WinitInputHelper) {
        
    }

    fn render(&mut self, ctx: &mut xyz::RenderContext) {
        ctx.clear();
        
        ctx.draw_line(CUBE[0], CUBE[1], 0xffffff);
        ctx.draw_line(CUBE[1], CUBE[2], 0xffffff);
        ctx.draw_line(CUBE[2], CUBE[3], 0xffffff);
        ctx.draw_line(CUBE[3], CUBE[0], 0xffffff);

        ctx.draw_line(CUBE[4], CUBE[5], 0xffffff);
        ctx.draw_line(CUBE[5], CUBE[6], 0xffffff);
        ctx.draw_line(CUBE[6], CUBE[7], 0xffffff);
        ctx.draw_line(CUBE[7], CUBE[4], 0xffffff);

        ctx.draw_line(CUBE[0], CUBE[4], 0xffffff);
        ctx.draw_line(CUBE[1], CUBE[5], 0xffffff);
        ctx.draw_line(CUBE[2], CUBE[6], 0xffffff);
        ctx.draw_line(CUBE[3], CUBE[7], 0xffffff);
    }

    fn gui(&mut self, ctx: &egui::Context) {
        
    }
}