#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragBaseColor;
layout(location = 4) flat in uint fragUseTexture;
layout(location = 5) in float fragDist;
layout(location = 6) in vec3 fragWorldPos;

layout(binding = 1) uniform LightUBO {
	vec3 pos;
	vec3 dir;
	vec3 color;
	mat4 lightViewProj;
} light;

layout(binding = 5) uniform sampler2D texSampler;
layout(binding = 7) uniform sampler2D shadowMap;

layout(location = 0) out vec4 outColor;

void main() {
	vec3 albedo;
	if (fragUseTexture != 0u) {
		vec4 tex = texture(texSampler, fragTexCoord);
		if (tex.a < 0.5) discard;
		albedo = tex.rgb;
	} else {
		albedo = fragBaseColor;
	}

	vec3 normal = normalize(fragNormal);
	vec3 lightDir = normalize(-light.dir);
	float nDiff = max(dot(normal, lightDir), 0.0);

	float shadow = 1.0;
	vec4 lightClip = light.lightViewProj * vec4(fragWorldPos, 1.0);
	vec3 projCoords = lightClip.xyz / lightClip.w;
	projCoords.xy = projCoords.xy * 0.5 + 0.5;

	if (projCoords.x >= 0.0 && projCoords.x <= 1.0 &&
		projCoords.y >= 0.0 && projCoords.y <= 1.0 &&
		projCoords.z >= 0.0 && projCoords.z <= 1.0) {
		float closestDepth = texture(shadowMap, projCoords.xy).r;
		float currentDepth = projCoords.z;
		float bias = max(0.005 * (1.0 - nDiff), 0.001);
		shadow = currentDepth - bias > closestDepth ? 0.3 : 1.0;
	}

	vec3 ambient = 0.1 * albedo;
	vec3 diffuse = shadow * nDiff * light.color * albedo;
	vec3 finalColor = ambient + diffuse;

	const vec3 fogColor = vec3(0.7, 0.75, 0.85);
	const float fogStart = 80.0;
	const float fogEnd = 300.0;
	float fogFactor = clamp((fragDist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
	finalColor = mix(finalColor, fogColor, fogFactor);

	outColor = vec4(finalColor, 1.0);
}
