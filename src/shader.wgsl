// Vertex shader
struct Camera {
    position: vec2<f32>,
    scale: f32,
    aspect: f32,
};

@group(0)
@binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct InstanceInput {
    @location(1) position: vec2<f32>,
    @location(2) radius: f32,
    @location(3) color: i32,
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
    
    let r = (instance.color >> 16u);
    let g = (instance.color >> 8u ) & 0xff;
    let b = (instance.color       ) & 0xff;
    out.color = vec3(f32(r), f32(g), f32(b)) / 255.0;

    let world_pos = vertex.position * instance.radius + instance.position;
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
