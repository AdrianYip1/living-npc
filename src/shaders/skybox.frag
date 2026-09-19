#version 450

layout(location = 0) in vec3 inDir;
layout(location = 1) in float outTime;
layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform samplerCube skyboxDay;
layout(binding = 2) uniform samplerCube skyboxNoon;
layout(binding = 3) uniform samplerCube skyboxNight;

void main() {
    vec4 skybox1 = texture(skyboxDay, inDir);
    vec4 skybox2 = texture(skyboxNoon, inDir);
    vec4 skybox3 = texture(skyboxNight, inDir);

    float timeProgress = mod(outTime, 3.0);

    if (timeProgress < 1.0) {
        fragColor = mix(skybox1, skybox2, timeProgress);
    } else if (timeProgress < 2.0) {
        fragColor = mix(skybox2, skybox3, timeProgress - 1.0);
    } else {
        fragColor = mix(skybox3, skybox1, timeProgress - 2.0);
    }
}
