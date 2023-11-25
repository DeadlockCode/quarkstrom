var<private> VERTICES: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2<f32>(-1.7321,-1.0),
    vec2<f32>( 1.7321,-1.0), // sqrt(3) ≈ 1.7321
    vec2<f32>( 0.0   , 2.0),
);

// Vertex shader
struct View {
    position: vec2<f32>,
    scale: f32,
    xy: u32,
};

@group(0)
@binding(0)
var<uniform> view: View;

struct VertexInput {
    @builtin(vertex_index) index: u32
};

struct InstanceInput {
    @location(0) position: vec2<f32>,
    @location(1) radius: f32,
    @location(2) color: u32,
};

struct VertexOutput {
    @builtin(position) clip_space: vec4<f32>,
    @location(0) local_space: vec2<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;

    let x = (view.xy       ) & 0xffffu;
    let y = (view.xy >> 16u) & 0xffffu;
    let aspect = f32(y) / f32(x);

    // Hexadecimal -> RGB 0-1
    let r = (instance.color >> 16u);
    let g = (instance.color >> 8u ) & 0xffu;
    let b = (instance.color       ) & 0xffu;
    let color = vec3(f32(r), f32(g), f32(b)) / 255.0;

    // Local space
    let local_space = VERTICES[vertex.index];

    // If the circle is smaller than a pixel, we need to snap it to the pixel grid and make it larger
    // This ensures that the circle never becomes invisible when zooming out or when the circle is very small
    var position = instance.position;
    var radius = instance.radius;
    if radius * f32(y) < 1.414214 * view.scale {
        let one = ((position - view.position) / view.scale);
        let two = (floor(one * f32(y)) + 0.5) / f32(y);
        let three = two * view.scale + view.position;
        position = three;
        radius = view.scale / f32(y);
    }

    // Object space -> World space
    let world_space = local_space * radius + position;

    // World space -> View space
    let view_space = (world_space - view.position) / view.scale;

    // View space -> Clip space
    let clip_space = vec4<f32>(view_space.x * aspect, view_space.y, 0.0, 1.0);

    // Return
    out.clip_space  = clip_space;
    out.local_space = local_space;
    out.color       = color;
    return out;
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if dot(in.local_space, in.local_space) > 1.0 {
        discard;
    }
    return vec4<f32>(in.color, 1.0);
}

// Version of fragment shader with lighting (diffuse + ambient)
//
// @fragment
// fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
//     if dot(in.local_space, in.local_space) > 1.0 {
//         discard;
//     }
//     let ambient = vec3<f32>(0.2, 0.5, 1.0);
//     let ambient_strength = 0.02;
//     let light_dir = vec3<f32>(1.0, 1.0, 2.0) / sqrt(6.0);
//     let x = in.local_space.x; let y = in.local_space.y;
//     let normal = vec3<f32>(x, y, sqrt(1.0 - x*x - y*y));
//     let brightness = (0.5 + max(dot(light_dir, normal), -0.5)) / 1.5;
//     return vec4<f32>(ambient * ambient_strength + in.color * ((1.0 - ambient_strength) * brightness * brightness * brightness), 1.0);
// }