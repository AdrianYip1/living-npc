#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(std430, set = 0, binding = 2) readonly buffer MorphDeltas {
    vec4 deltas[];
} morph;

layout(std430, set = 0, binding = 3) readonly buffer WeightsBuffer {
    float weights[];
} weightBuffer;

layout(push_constant) uniform MorphPush {
    mat4 model;
    uint morphStartIndex;
    uint targetCount;
    uint vertexOffset;
    uint vertexCount;
    uint weightsStartIndex;
    uint weightBase;
} PushConstants;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;

void main() {
    vec3 pos = inPosition;

    uint localVert = gl_VertexIndex - PushConstants.vertexOffset;
    for (uint t = 0; t < PushConstants.targetCount; t++) {
        uint weightIndex = PushConstants.weightBase + PushConstants.weightsStartIndex + t;
        float w = weightBuffer.weights[weightIndex];
        if (w == 0.0) continue;
        uint index = PushConstants.morphStartIndex + t * PushConstants.vertexCount + localVert;
        pos += w * morph.deltas[index].xyz;
    }

    gl_Position = ubo.proj * ubo.view * PushConstants.model * vec4(pos, 1.0);
    fragColor = inColor;
    fragNormal = mat3(PushConstants.model) * inNormal;
}
