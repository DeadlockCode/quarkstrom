use std::f32::consts::TAU;

use egui::{mutex::Mutex, Color32, Response, Ui};
use fastrand;
use quarkstrom::{self, run, Config};
use ultraviolet::Vec2;
use winit::dpi::PhysicalSize;

use once_cell::sync::Lazy;

static PARTICLES: Lazy<Mutex<Option<Vec<Particle>>>> = Lazy::new(|| Mutex::new(None));
static GUI: Lazy<Mutex<Vec<Gui>>> = Lazy::new(|| Mutex::new(Vec::new()));

fn main() {
    let config = Config {
        tps_cap: Some(60),
        view_size: 512.0,
        window_size: PhysicalSize::new(1280, 720),
    };
    run::<Simulation, Gui>(config);
}

#[derive(Clone)]
enum Boundary {
    Square(f32),
    ReflectedCircle(f32),
    InverseCircle(f32),
    None,
}

#[derive(Clone)]
struct Gui {
    pub restart: bool,
    pub boundary: Boundary,
    pub num_particles: usize,
    pub types: u32,
    pub attraction_matrix: Vec<f32>,
    pub simulation_speed: f32,
    pub velocity_half_life: f32,

    texture: Option<egui::TextureHandle>,
}

impl quarkstrom::Gui for Gui {
    fn new() -> Self {
        Self {
            restart: false,
            boundary: Boundary::Square(500.0),
            num_particles: 500,
            types: 5,
            attraction_matrix: vec![0.; 25],
            simulation_speed: 1.0,
            velocity_half_life: 0.05,
            texture: None,
        }
    }

    fn gui(&mut self, ctx: &egui::Context) {
        fn hue2rgb(c: f32) -> [u8; 4] {
            return [
                ((((c + 1.0).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0) * 255.0) as u8,
                ((((c + 2.0 / 3.0).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0) * 255.0) as u8,
                ((((c + 1.0 / 3.0).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0) * 255.0) as u8,
                (1.0 * 255.0) as u8,
            ];
        }
        let texture = self.texture.get_or_insert_with(|| {
            ctx.load_texture(
                "Square",
                egui::ColorImage::new([1, 1], egui::Color32::WHITE),
                Default::default(),
            )
        });

        egui::Window::new("Quarkstrom").show(&ctx, |ui| {
            ui.style_mut().spacing.slider_width = 300.0;

            self.restart = ui.button("Restart").clicked();
            ui.separator();

            let boundary_idx = match self.boundary {
                Boundary::Square(_) => 0,
                Boundary::ReflectedCircle(_) => 1,
                Boundary::InverseCircle(_) => 2,
                Boundary::None => 3,
            };

            let items = ["Square", "Reflected Circle", "Inverse Circle", "None"];

            ui.horizontal(|ui| {
                for i in 0..4 {
                    if ui.selectable_label(i == boundary_idx, items[i]).clicked() {
                        let l = match self.boundary {
                            Boundary::Square(s) => s / 2.0,
                            Boundary::ReflectedCircle(r) => r,
                            Boundary::InverseCircle(r) => r,
                            Boundary::None => 250.0,
                        };
                        match i {
                            0 => self.boundary = Boundary::Square(2.0 * l),
                            1 => self.boundary = Boundary::ReflectedCircle(l),
                            2 => self.boundary = Boundary::InverseCircle(l),
                            3 => self.boundary = Boundary::None,
                            _ => {}
                        }
                    }
                }
            });

            match &mut self.boundary {
                Boundary::Square(side) => {
                    ui.horizontal(|ui| {
                        let id = ui.label("Side: ").layer_id.id;
                        ui.add(egui::Slider::new(side, 0.0..=1000.0))
                            .labelled_by(id);
                    });
                }
                Boundary::InverseCircle(radius) | Boundary::ReflectedCircle(radius) => {
                    ui.horizontal(|ui| {
                        let id = ui.label("Side: ").layer_id.id;
                        ui.add(egui::Slider::new(radius, 0.0..=500.0))
                            .labelled_by(id);
                    });
                }
                Boundary::None => {}
            }

            ui.horizontal(|ui| {
                let id = ui.label("Simulation Speed: ").layer_id.id;
                ui.add(egui::Slider::new(&mut self.simulation_speed, 0.0..=10.0))
                    .labelled_by(id);
            });

            ui.horizontal(|ui| {
                let id = ui.label("Velocity Half Life: ").layer_id.id;
                ui.add(egui::Slider::new(&mut self.velocity_half_life, 0.0..=1.0))
                    .on_hover_text("The amount of time until the velocity halves itself.")
                    .labelled_by(id);
            });

            ui.horizontal(|ui| {
                let id = ui.label("Particles: ").layer_id.id;
                ui.add(egui::Slider::new(&mut self.num_particles, 0..=64 * 256))
                    .labelled_by(id);
            });

            let old_types = self.types;
            ui.horizontal(|ui| {
                let id = ui.label("Colors: ").layer_id.id;
                if ui
                    .add(egui::Slider::new(&mut self.types, 1..=256))
                    .labelled_by(id)
                    .changed()
                {
                    let last_matrix = self.attraction_matrix.clone();

                    self.attraction_matrix = vec![0f32; self.types as usize * self.types as usize];

                    for i in 0..old_types.min(self.types) {
                        for j in 0..old_types.min(self.types) {
                            self.attraction_matrix[(i + j * self.types) as usize] =
                                last_matrix[(i + j * old_types) as usize];
                        }
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("Attraction Matric:");
                if ui.button("Reset").clicked() {
                    self.attraction_matrix = vec![0.0; (self.types * self.types) as usize];
                }
                if ui.button("Randomize").clicked() {
                    for i in 0..self.types * self.types {
                        self.attraction_matrix[i as usize] = fastrand::f32() * 2.0 - 1.0;
                    }
                }
                if ui.button("Snake").clicked() {
                    self.attraction_matrix = vec![0f32; self.types as usize * self.types as usize];
                    for i in 0..self.types {
                        if i != 0 {
                            self.attraction_matrix[(i + (i - 1) * self.types) as usize] = 0.2;
                        }
                        self.attraction_matrix[(i + i * self.types) as usize] = 1.0;
                    }
                }
            });

            fn color(ui: &mut Ui, texture: &egui::TextureHandle, color: [u8; 4]) -> Response {
                ui.add(
                    egui::ImageButton::new(texture.id(), [32.0, 32.0])
                        .tint(Color32::from_rgb(color[0], color[1], color[2])),
                )
            }

            ui.horizontal(|ui| {
                color(ui, &texture, [0; 4]);
                for i in 0..self.types as usize {
                    color(ui, &texture, hue2rgb(i as f32 / self.types as f32));
                }
            });
            for j in 0..self.types as usize {
                ui.horizontal(|ui| {
                    color(ui, &texture, hue2rgb(j as f32 / self.types as f32));
                    for i in 0..self.types as usize {
                        ui.add(
                            egui::DragValue::new(
                                &mut self.attraction_matrix[i + j * self.types as usize],
                            )
                            .speed(0.01)
                            .fixed_decimals(3),
                        );
                    }
                });
            }
        });

        GUI.lock().push(self.clone());
    }

    fn render(&mut self) -> (Option<Vec<quarkstrom::Instance>>, Option<Vec<quarkstrom::Vertex>>) {
        if let Some(particles) = PARTICLES.lock().clone() {
            let mut result = particles.iter()
                .map(|particle| quarkstrom::Instance {
                    position: particle.pos,
                    radius: 1.0,
                    color: hue2rgb(particle.typ as f32 / self.types as f32),
                })
                .collect::<Vec<_>>();

            for i in 0..result.len() {
                match self.boundary {
                    Boundary::Square(side) => {
                        for x in -1..=1 {
                            for y in -1..=1 {
                                if x == 0 && y == 0 {
                                    continue;
                                }
                                result.push(quarkstrom::Instance {
                                    position: result[i].position
                                        + Vec2::new(side * x as f32, side * y as f32),
                                    radius: 1.0,
                                    color: 0x888888,
                                });
                            }
                        }
                    }
                    Boundary::ReflectedCircle(radius) => {
                        if result[i].position != Vec2::zero() {
                            result.push(quarkstrom::Instance {
                                position: particles[i].pos
                                    * (1.0 - particles[i].pos.mag().recip() * (2.0 * radius)),
                                radius: 1.0,
                                color: 0x888888,
                            });
                        }
                    }
                    Boundary::InverseCircle(radius) => {
                        let mag_sq = result[i].position.mag_sq();
                        if mag_sq > 0.1 * radius {
                            result.push(quarkstrom::Instance {
                                position: -result[i].position / mag_sq * radius * radius,
                                radius: 1.0 / mag_sq * radius * radius,
                                color: 0x888888,
                            });
                        }
                    }
                    Boundary::None => {}
                }
            }

            return (Some(result), None);
        }
        return (None, None);
        
    }
}

struct World {
    boundary: Boundary,
    types: u32,
    #[allow(dead_code)]
    universal_repulsive_strength: f32,
    interaction_distance: f32,
    interaction_multiplier: f32,
    velocity_half_life: f32,
    dt: f32,
}

#[derive(Clone)]
struct Particle {
    pos: Vec2,
    vel: Vec2,
    acc: Vec2,
    typ: u32,
}

struct Simulation {
    world: World,
    particles: Vec<Particle>,
    attraction_matrix: Vec<f32>,
}

impl Simulation {
    fn attraction(&self, i: usize, j: usize) -> f32 {
        return self.attraction_matrix
            [(self.particles[i].typ + self.particles[j].typ * self.world.types) as usize];
    }

    fn force(&self, r: f32, a: f32) -> f32 {
        let b = 0.3;
        if r < b {
            return r / b - 1.;
        } else if b < r && r < 1. {
            return a * (1. - (2. * r - 1. - b).abs() / (1. - b));
        } else {
            return 0.;
        }
    }

    pub fn restart(&mut self, num_particles: usize) {
        self.particles = (0..num_particles)
            .map(|_| {
                let r = fastrand::f32();
                let a = fastrand::f32() * TAU;
                let d = match self.world.boundary {
                    Boundary::Square(s) => s * 0.5,
                    Boundary::ReflectedCircle(r) | Boundary::InverseCircle(r) => r,
                    Boundary::None => 500.0,
                };

                let typ = fastrand::u32(0..self.world.types);

                Particle {
                    pos: d * r.sqrt() * Vec2::new(a.cos(), a.sin()),
                    vel: Vec2::zero(),
                    acc: Vec2::zero(),
                    typ,
                }
            })
            .collect::<Vec<_>>();
    }

    fn gui(&mut self) {
        while let Some(gui) = GUI.lock().pop() {
            self.world.dt = 0.01 * gui.simulation_speed;
            self.world.velocity_half_life = gui.velocity_half_life;

            self.world.boundary = gui.boundary;
            if gui.types < self.world.types {
                for particle in &mut self.particles {
                    if particle.typ >= gui.types {
                        particle.typ = fastrand::u32(0..gui.types);
                    }
                }
            }
            self.world.types = gui.types;
            self.attraction_matrix = gui.attraction_matrix.clone();

            let len = self.particles.len();
            if gui.num_particles > len {
                for _ in len..gui.num_particles {
                    let r = fastrand::f32();
                    let a = fastrand::f32() * TAU;
                    let d = match self.world.boundary {
                        Boundary::Square(s) => s * 0.5,
                        Boundary::ReflectedCircle(r) | Boundary::InverseCircle(r) => r,
                        Boundary::None => 500.0,
                    };

                    let typ = fastrand::u32(0..self.world.types);

                    self.particles.push(Particle {
                        pos: d * r.sqrt() * Vec2::new(a.cos(), a.sin()),
                        vel: Vec2::zero(),
                        acc: Vec2::zero(),
                        typ,
                    });
                }
            } else if gui.num_particles < len {
                for _ in gui.num_particles..len {
                    self.particles.pop();
                }
            }

            if gui.restart {
                self.restart(gui.num_particles);
            }

            for particle in &mut self.particles {
                fn modulo(lhs: f32, rhs: f32) -> f32 {
                    (lhs % rhs + rhs) % rhs
                }

                let drag = 0.5f32.powf(self.world.dt / self.world.velocity_half_life);
                particle.vel = drag * particle.vel + particle.acc * self.world.dt;
                particle.pos = particle.pos + particle.vel * self.world.dt;

                match self.world.boundary {
                    Boundary::Square(side) => {
                        particle.pos.x = modulo(particle.pos.x + side * 0.5, side) - side * 0.5;
                        particle.pos.y = modulo(particle.pos.y + side * 0.5, side) - side * 0.5;
                    }
                    Boundary::ReflectedCircle(radius) => {
                        let mag_sq = particle.pos.mag_sq();
                        if mag_sq > radius * radius {
                            let n = particle.pos;
                            particle.pos = n * (1.0 - mag_sq.sqrt().recip() * (2.0 * radius));
                        }
                    }
                    Boundary::InverseCircle(radius) => {
                        if particle.pos.mag_sq() > radius * radius {
                            particle.vel = (-particle.vel).reflected(particle.pos.normalized());

                            let n = particle.pos / radius;
                            particle.pos = -n / n.mag_sq() * radius;
                        }
                    }
                    Boundary::None => {}
                }
            }
        }
    }
}

impl quarkstrom::Simulation<Gui> for Simulation {
    fn new() -> Self {
        let world = World {
            boundary: Boundary::Square(500.0),
            types: 0,
            universal_repulsive_strength: 50.0,
            interaction_distance: 100.0,
            interaction_multiplier: 1.0,
            velocity_half_life: 0.04,
            dt: 0.01,
        };

        let particles = Vec::new();

        let attraction_matrix = Vec::new();

        Self {
            particles,
            world,
            attraction_matrix,
        }
    }

    fn update(&mut self) {
        self.gui();

        match self.world.boundary {
            Boundary::Square(width) => {
                let offset = Vec2::new(width, width) * 0.5;

                let mut grid = Grid::new(self.world.interaction_distance, width);

                for i in 0..self.particles.len() {
                    self.particles[i].acc = Vec2::zero();
                    grid.insert(i, self.particles[i].pos + offset);
                }

                for i in 0..self.particles.len() {
                    for &(j, offset) in grid.get(self.particles[i].pos + offset) {
                        if i == j {
                            continue;
                        }
                        let d = self.particles[j].pos - self.particles[i].pos + offset;
                        let r = d.mag();

                        let f =
                            self.force(r / self.world.interaction_distance, self.attraction(i, j));

                        let acc = d
                            * (self.world.interaction_distance * self.world.interaction_multiplier
                                / r);
                        self.particles[i].acc += acc * f;
                    }
                }
            }
            Boundary::ReflectedCircle(radius) => {
                for i in 0..self.particles.len() {
                    self.particles[i].acc = Vec2::zero();
                    for j in 0..i {
                        let d = self.particles[j].pos - self.particles[i].pos;
                        let r = d.mag();

                        let f1 =
                            self.force(r / self.world.interaction_distance, self.attraction(i, j));
                        let f2 =
                            self.force(r / self.world.interaction_distance, self.attraction(j, i));

                        let acc = d
                            * (self.world.interaction_distance * self.world.interaction_multiplier
                                / r);
                        self.particles[i].acc += acc * f1;
                        self.particles[j].acc -= acc * f2;

                        if self.particles[j].pos != Vec2::zero() {
                            let reflect = self.particles[j].pos
                                * (1.0 - self.particles[j].pos.mag().recip() * (2.0 * radius));
                            let d = reflect - self.particles[i].pos;
                            let r = d.mag();

                            let f = self
                                .force(r / self.world.interaction_distance, self.attraction(i, j));

                            self.particles[i].acc += d
                                * (self.world.interaction_distance
                                    * self.world.interaction_multiplier
                                    * f
                                    / r);
                        }

                        if self.particles[i].pos != Vec2::zero() {
                            let reflect = self.particles[i].pos
                                * (1.0 - self.particles[i].pos.mag().recip() * (2.0 * radius));
                            let d = reflect - self.particles[j].pos;
                            let r = d.mag();

                            let f = self
                                .force(r / self.world.interaction_distance, self.attraction(j, i));

                            self.particles[j].acc += d
                                * (self.world.interaction_distance
                                    * self.world.interaction_multiplier
                                    * f
                                    / r);
                        }
                    }
                }
            }
            Boundary::InverseCircle(radius) => {
                for i in 0..self.particles.len() {
                    self.particles[i].acc = Vec2::zero();
                    for j in 0..i {
                        let d = self.particles[j].pos - self.particles[i].pos;
                        let r = d.mag();

                        let f1 =
                            self.force(r / self.world.interaction_distance, self.attraction(i, j));
                        let f2 =
                            self.force(r / self.world.interaction_distance, self.attraction(j, i));

                        let acc = d
                            * (self.world.interaction_distance * self.world.interaction_multiplier
                                / r);
                        self.particles[i].acc += acc * f1;
                        self.particles[j].acc -= acc * f2;

                        if self.particles[j].pos != Vec2::zero() {
                            let mag_sq = self.particles[j].pos.mag_sq();
                            let hyper = -self.particles[j].pos / mag_sq * radius * radius;
                            let d = hyper - self.particles[i].pos;
                            let r = d.mag();

                            let f = self
                                .force(r / self.world.interaction_distance, self.attraction(i, j));

                            self.particles[i].acc += d
                                * (self.world.interaction_distance
                                    * self.world.interaction_multiplier
                                    * f
                                    / r);
                        }

                        if self.particles[i].pos != Vec2::zero() {
                            let mag_sq = self.particles[i].pos.mag_sq();
                            let hyper = -self.particles[i].pos / mag_sq * radius * radius;
                            let d = hyper - self.particles[j].pos;
                            let r = d.mag();

                            let f = self
                                .force(r / self.world.interaction_distance, self.attraction(j, i));

                            self.particles[j].acc += d
                                * (self.world.interaction_distance
                                    * self.world.interaction_multiplier
                                    * f
                                    / r);
                        }
                    }
                }
            }
            Boundary::None => {
                for i in 0..self.particles.len() {
                    self.particles[i].acc = Vec2::zero();
                    for j in 0..i {
                        let d = self.particles[j].pos - self.particles[i].pos;
                        let r = d.mag();

                        let f1 =
                            self.force(r / self.world.interaction_distance, self.attraction(i, j));
                        let f2 =
                            self.force(r / self.world.interaction_distance, self.attraction(j, i));

                        let acc = d
                            * (self.world.interaction_distance * self.world.interaction_multiplier
                                / r);
                        self.particles[i].acc += acc * f1;
                        self.particles[j].acc -= acc * f2;
                    }
                }
            }
        }

        for particle in &mut self.particles {
            fn modulo(lhs: f32, rhs: f32) -> f32 {
                (lhs % rhs + rhs) % rhs
            }

            let drag = 0.5f32.powf(self.world.dt / self.world.velocity_half_life);
            particle.vel = drag * particle.vel + particle.acc * self.world.dt;
            particle.pos = particle.pos + particle.vel * self.world.dt;

            match self.world.boundary {
                Boundary::Square(side) => {
                    particle.pos.x = modulo(particle.pos.x + side * 0.5, side) - side * 0.5;
                    particle.pos.y = modulo(particle.pos.y + side * 0.5, side) - side * 0.5;
                }
                Boundary::ReflectedCircle(radius) => {
                    let mag_sq = particle.pos.mag_sq();
                    if mag_sq > radius * radius {
                        let n = particle.pos;
                        particle.pos = n * (1.0 - mag_sq.sqrt().recip() * (2.0 * radius));
                    }
                }
                Boundary::InverseCircle(radius) => {
                    if particle.pos.mag_sq() > radius * radius {
                        particle.vel = (-particle.vel).reflected(particle.pos.normalized());

                        let n = particle.pos / radius;
                        particle.pos = -n / n.mag_sq() * radius;
                    }
                }
                Boundary::None => {}
            }
        }
    }

    fn convert(&mut self) {
        *PARTICLES.lock() = Some(self.particles.clone());
    }
}

fn hue2rgb(hue: f32) -> u32 {
    let x: f32 = 1.0 - (((hue * 6.0) % 2.0) - 1.0).abs();
    let i = (hue * 6.0).floor() as u32;
    let (r, g, b) = match i {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        5 => (1.0, 0.0, x),
        _ => unreachable!(),
    };
    ((r * 255.0) as u32) << 16 | ((g * 255.0) as u32) << 8 | (b * 255.0) as u32
}

struct Grid {
    cells: Vec<Vec<(usize, Vec2)>>,
    resolution: usize,
    width: f32,
}

impl Grid {
    fn new(min_cell_width: f32, width: f32) -> Self {
        let resolution = ((width / min_cell_width).floor() as usize).max(1);

        Self {
            cells: vec![Vec::new(); resolution * resolution],
            resolution,
            width,
        }
    }

    fn insert(&mut self, index: usize, position: Vec2) {
        let pos = position / self.width * self.resolution as f32;
        let x = pos.x.floor() as usize;
        let y = pos.y.floor() as usize;

        for i in 0..=2 {
            for j in 0..=2 {
                let (mut x, mut y) = (x + i - 1, y + j - 1);
                let mut offset = Vec2::zero();
                if x == usize::MAX {
                    x = self.resolution - 1;
                    offset.x = self.width;
                }
                if y == usize::MAX {
                    y = self.resolution - 1;
                    offset.y = self.width;
                }
                if x == self.resolution {
                    x = 0;
                    offset.x = -self.width;
                }
                if y == self.resolution {
                    y = 0;
                    offset.y = -self.width;
                }
                self.cells[x + y * self.resolution].push((index, offset));
            }
        }
    }

    fn get(&mut self, position: Vec2) -> &[(usize, Vec2)] {
        let pos = position / self.width * self.resolution as f32;
        let x = pos.x.floor() as usize;
        let y = pos.y.floor() as usize;

        &self.cells[x + y * self.resolution]
    }
}
