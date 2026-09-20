#version 450

layout(location = 0) in vec3 inDir;
layout(location = 1) in float dayFraction;
layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform samplerCube skyboxDay;
layout(binding = 2) uniform samplerCube skyboxNoon;
layout(binding = 3) uniform samplerCube skyboxNight;

void main() {
    vec4 day   = texture(skyboxDay, inDir);
    vec4 noon  = texture(skyboxNoon, inDir);
    vec4 night = texture(skyboxNight, inDir);

    float t = fract(dayFraction);

    // 0.0 = midnight, 0.25 = 6am, 0.5 = noon, 0.75 = 6pm
    vec4 sky;
    vec3 tint;
    float brightness;

    if (t < 0.2) {
        // night
        sky = night;
        tint = vec3(0.6, 0.65, 0.8);
        brightness = 0.3;
    } else if (t < 0.3) {
        // sunrise transition
        float s = (t - 0.2) / 0.1;
        sky = mix(night, day, s);
        tint = mix(vec3(0.6, 0.65, 0.8), vec3(1.0, 0.7, 0.5), s * (1.0 - s) * 4.0)
             + vec3(1.0, 1.0, 1.0) * s;
        tint = normalize(tint) * length(mix(vec3(0.6, 0.65, 0.8), vec3(1.0, 1.0, 1.0), s));
        brightness = mix(0.3, 1.0, s);
    } else if (t < 0.7) {
        // daytime
        sky = mix(day, noon, smoothstep(0.3, 0.5, t) - smoothstep(0.5, 0.7, t));
        tint = vec3(1.0);
        brightness = 1.0;
    } else if (t < 0.8) {
        // sunset transition
        float s = (t - 0.7) / 0.1;
        sky = mix(day, night, s);
        tint = mix(vec3(1.0, 1.0, 1.0), vec3(0.6, 0.65, 0.8), s)
             + vec3(0.4, 0.15, 0.0) * s * (1.0 - s) * 4.0;
        brightness = mix(1.0, 0.3, s);
    } else {
        // night
        sky = night;
        tint = vec3(0.6, 0.65, 0.8);
        brightness = 0.3;
    }

    fragColor = vec4(sky.rgb * tint * brightness, 1.0);
}
