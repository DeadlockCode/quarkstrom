struct View {
    position: vec2<f32>,
    scale: f32,
    aspect: f32,
};

@group(0)
@binding(0)
var<uniform> view: View;

struct VertexOutput {
    @builtin(position) clip_space: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec2<f32>,
    @location(1) color: u32,
) -> VertexOutput {
    var result: VertexOutput;

    let r = (color >> 16u);
    let g = (color >> 8u ) & 0xffu;
    let b = (color       ) & 0xffu;
    let color = vec3(f32(r), f32(g), f32(b)) / 255.0;

    let view_space = (position - view.position) / view.scale;
    let clip_space = vec4<f32>(view_space.x * view.aspect, view_space.y, 0.0, 1.0);

    result.clip_space = clip_space;
    result.color = color;
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(vertex.color, 1.0);
}