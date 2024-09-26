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
    @location(0) min: vec2<f32>,
    @location(1) max: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_space: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var result: VertexOutput;

    let x = (view.xy       ) & 0xffffu;
    let y = (view.xy >> 16u) & 0xffffu;
    let aspect = f32(y) / f32(x);

    let local_space = vec2<f32>(f32(vertex.index & 1u), f32(vertex.index >> 1u));
    let position = instance.min + local_space * (instance.max - instance.min);

    let view_space = (position - view.position) / view.scale;
    let clip_space = vec4<f32>(view_space.x * aspect, view_space.y, 0.0, 1.0);

    result.clip_space = clip_space;
    result.color = vec4(pow(instance.color.rgb, vec3(2.2)), instance.color.a);
    return result;
}

@fragment
fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
    return vertex.color;
}