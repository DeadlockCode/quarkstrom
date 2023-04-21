// Global

struct World {
    size: f32,
    num_particles: u32,
    num_colors: u32,
    universal_repulsive_strength: f32,
    repulsive_distance: f32,
    interaction_distance: f32,
    interaction_multiplier: f32,
    velocity_half_life: f32,
    dt: f32,
};

@group(0)
@binding(0)
var<uniform> world: World;

// Compute shader

struct Particle {
    px: f32,
    py: f32,
    vx: f32,
    vy: f32,
    col: u32,
}

@group(1)
@binding(0)
var<storage, read_write> particles: array<Particle>;

@group(2)
@binding(0)
var<storage, read> attraction_matrix: array<f32>;

struct ComputeInput {
    @builtin(global_invocation_id) global_id: vec3<u32>,
}

fn attraction(i: u32, j: u32) -> f32 {
    return attraction_matrix[particles[i].col + particles[j].col * world.num_colors];
}

fn force(r: f32, a: f32) -> f32 {
    let b = 0.3;
    if r < b {
        return r / b - 1.;
    }
    else if b < r && r < 1. {
        return a * (1.- abs(2.* r - 1.- b) / (1.- b));
    }
    else {
        return 0.;
    }
}

fn acc(i: u32) -> vec2<f32> {
    var result: vec2<f32>;
    
    for (var j = 0u; j < world.num_particles; j++) {
        if j == i { continue; }

        for (var x = -1; x <= 1; x++) {
            for (var y = -1; y <= 1; y++) {
                let dx = particles[j].px - particles[i].px + world.size * f32(x);
                let dy = particles[j].py - particles[i].py + world.size * f32(y);
                let mag_sq = dx * dx + dy * dy;
                let r = sqrt(mag_sq);

                var f = force(r / world.interaction_distance, attraction(i, j));

                result += vec2(dx, dy) * (world.interaction_multiplier * f / r);
            }
        }
    }

    return world.interaction_distance * result;
}

fn modf(a: f32, b: f32) -> f32 {
    return a-floor(a/b)*b;
}

@compute
@workgroup_size(64)
fn cs_main(
    compute: ComputeInput,
) {
    let i = compute.global_id.x;
    if i >= world.num_particles { return; }

    let acc = acc(i);
    let drag = pow(0.5, world.dt / world.velocity_half_life);
    particles[i].vx = drag * particles[i].vx + acc.x * world.dt;
    particles[i].vy = drag * particles[i].vy + acc.y * world.dt;
    particles[i].px = modf(particles[i].px + particles[i].vx * world.dt, world.size);
    particles[i].py = modf(particles[i].py + particles[i].vy * world.dt, world.size);
}

// Vertex shader

struct Camera {
    position: vec2<f32>,
    scale: f32,
    aspect: f32,
};

@group(1)
@binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct InstanceInput {
    @location(1) position: vec2<f32>,
    @location(2) velocity: vec2<f32>,
    @location(3) color: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
};

fn hue2rgb(c: f32) -> vec3<f32> {
    let k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c + k.xyz) * 6.0 - k.www);
    return clamp(p - k.xxx, vec3(0.0), vec3(1.0));
}

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    out.color = hue2rgb(f32(instance.color) / f32(world.num_colors));

    let world_pos = vertex.position + instance.position;
    var camera_pos = (world_pos - camera.position) / camera.scale;
    camera_pos.y *= camera.aspect;
    out.clip_position = vec4<f32>(camera_pos, 0.0, 1.0);
    out.uv = vertex.position;
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if dot(in.uv, in.uv) > 1.0 {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}
