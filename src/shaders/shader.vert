#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uvec4 inJoints;
layout(location = 5) in vec4 inWeights;

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

layout(std430, set = 0, binding = 4) readonly buffer PaletteBuffer {
    mat4 palette[];
};

layout(push_constant) uniform MorphPush {
    mat4 model;
    uint morphStartIndex;
    uint targetCount;
    uint vertexOffset;
    uint vertexCount;
    uint weightsStartIndex;
    uint weightBase;
    uint isSkinned;
    uint jointBase;
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

    vec3 normal = inNormal;

    if (PushConstants.isSkinned == 1u) {
        uint jb = PushConstants.jointBase;
        mat4 skin = inWeights.x * palette[jb + inJoints.x]
                  + inWeights.y * palette[jb + inJoints.y]
                  + inWeights.z * palette[jb + inJoints.z]
                  + inWeights.w * palette[jb + inJoints.w];
        pos = vec3(skin * vec4(pos, 1.0));
        normal = mat3(skin) * inNormal;
    }

    gl_Position = ubo.proj * ubo.view * PushConstants.model * vec4(pos, 1.0);
    fragColor = inColor;
    fragNormal = mat3(PushConstants.model) * normal;
}
