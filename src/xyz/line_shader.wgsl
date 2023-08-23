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

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) color: u32,
) -> VertexOutput {
    var result: VertexOutput;

    let r = (color >> 16u);
    let g = (color >> 8u ) & 0xffu;
    let b = (color       ) & 0xffu;
    let color = vec3(f32(r), f32(g), f32(b)) / 255.0;

    var view_space = position - vec3<f32>(view.x, view.y, view.z);

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

    let z = view_space.z;

    view_space.x *= view.aspect * view.factor;
    view_space.y *= view.factor;
    view_space.z = (view.zfar * view_space.z - view.zfar * view.znear) / (view.zfar - view.znear);

    let position = vec4<f32>(view_space, z);

    result.position = position;
    result.color = color;
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(vertex.color, 1.0);
}