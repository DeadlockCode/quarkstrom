var<private> VERTICES: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.7321,-1.0),
    vec2<f32>( 1.7321,-1.0), // sqrt(3) ≈ 1.7321
    vec2<f32>( 0.0   , 2.0),
);

// Vertex shader
struct View {
    x: f32,
    y: f32,
    z: f32,
    yrot: f32,
    xrot: f32,
    aspect: f32,
    factor: f32,
    zfar: f32,
    znear: f32
}

@group(0)
@binding(0)
var<uniform> view: View;

struct VertexInput {
    @builtin(vertex_index) index: u32
};

struct InstanceInput {
    @location(0) position: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) local_space: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;

    // Hexadecimal -> RGB 0-1
    let r = (instance.color >> 16u);
    let g = (instance.color >> 8u ) & 0xffu;
    let b = (instance.color       ) & 0xffu;
    let color = vec3(f32(r), f32(g), f32(b)) / 255.0;

    // Local space
    let local_space = VERTICES[vertex.index];

    // Object space -> World space
    let world_space = instance.position;

    // World space -> View space
    var view_space = world_space - vec3<f32>(view.x, view.y, view.z);

    view_space = vec3<f32>(
        view_space.x * cos(-view.yrot) + view_space.z * sin(-view.yrot),
        view_space.y,
        view_space.z * cos(-view.yrot) - view_space.x * sin(-view.yrot)
    );

    view_space = vec3<f32>(
        view_space.x,
        view_space.y * cos(-view.xrot) - view_space.z * sin(-view.xrot),
        view_space.z * cos(-view.xrot) + view_space.y * sin(-view.xrot)
    );

    view_space += vec3<f32>(local_space * instance.radius, 0.0);

    let z = view_space.z;

    view_space.x *= view.aspect * view.factor;
    view_space.y *= view.factor;
    view_space.z = (view.zfar * view_space.z - view.zfar * view.znear) / (view.zfar - view.znear);

    let position = vec4<f32>(view_space, z);

    // Return
    out.position  = position;
    out.local_space = local_space;
    out.color       = color;
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Outside circle?
    if dot(in.local_space, in.local_space) > 1.0 {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}