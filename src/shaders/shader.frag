#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragBaseColor;
layout(location = 4) flat in uint fragUseTexture;

layout(binding = 1) uniform LightUBO {
	vec3 pos;
	vec3 dir;
	vec3 color;
} light;

layout(binding = 5) uniform sampler2D texSampler;

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

	vec3 ambient = 0.1 * albedo;
	vec3 diffuse = nDiff * light.color * albedo;
	vec3 finalColor = ambient + diffuse;

	outColor = vec4(finalColor, 1.0);
}
