#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;

layout(binding = 1) uniform LightUBO {
	vec3 pos;
	vec3 dir;
	vec3 color;
} light;

layout(location = 0) out vec4 outColor;

void main() {
	vec3 normal = normalize(fragNormal);
	vec3 lightDir = normalize(-light.dir);
	float nDiff = max(dot(normal, lightDir), 0.0);

	vec3 ambient = 0.1 * fragColor;
	vec3 diffuse = nDiff * light.color * fragColor;
	vec3 finalColor = ambient + diffuse;

	outColor = vec4(finalColor, 1.0);
}
