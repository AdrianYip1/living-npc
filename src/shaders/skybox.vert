#version 450

layout(binding = 0) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    float time;
} ubo;

layout(location = 0) out vec3 outDir;
layout(location = 1) out float outTime;

void main() {
    vec2 pos = vec2(gl_VertexIndex == 1 ? 3.0 : -1.0,
                    gl_VertexIndex == 2 ? -3.0 : 1.0);

    vec4 viewPos = inverse(ubo.proj) * vec4(pos, 1.0, 1.0);
    outDir = transpose(mat3(ubo.view)) * normalize(viewPos.xyz / viewPos.w);

    outTime = ubo.time;
    gl_Position = vec4(pos, 1.0, 1.0);
}
