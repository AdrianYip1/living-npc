#include "modelLoading.hpp"
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include <enginemath/vec4.hpp>
#include <enginemath/mat4.hpp>

#include <cstring>
#include <cmath>
#include <string>
#include <stdexcept>

static enginemath::Mat4 quatToMat(float x, float y, float z, float w) {
	float xx = x * x, yy = y * y, zz = z * z;
	float xy = x * y, xz = x * z, yz = y * z;
	float wx = w * x, wy = w * y, wz = w * z;
	return enginemath::Mat4(
		enginemath::Vec4(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz),         2.0f * (xz - wy),         0.0f),
		enginemath::Vec4(2.0f * (xy - wz),         1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx),         0.0f),
		enginemath::Vec4(2.0f * (xz + wy),         2.0f * (yz - wx),         1.0f - 2.0f * (xx + yy), 0.0f),
		enginemath::Vec4(0.0f, 0.0f, 0.0f, 1.0f)
	);
}

static void slerp(float out[4], const float a[4], const float b0[4], float t) {
	float b[4] = { b0[0], b0[1], b0[2], b0[3] };
	float dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
	if (dot < 0.0f) { for (int k = 0; k < 4; k++) b[k] = -b[k]; dot = -dot; }

	if (dot > 0.9995f) {
		float len = 0.0f;
		for (int k = 0; k < 4; k++) { out[k] = a[k] + (b[k] - a[k]) * t; len += out[k]*out[k]; }
		len = std::sqrt(len);
		for (int k = 0; k < 4; k++) out[k] /= len;
		return;
	}

	float th0 = std::acos(dot), th = th0 * t;
	float s0 = std::sin(th0 - th) / std::sin(th0), s1 = std::sin(th) / std::sin(th0);
	for (int k = 0; k < 4; k++) out[k] = a[k]*s0 + b[k]*s1;
}

bool HTN::Loader::loadModel(const std::string& modelPath, fModel& out, Skeleton& skeleton) {
	cgltf_options options = {};
	cgltf_data* data = nullptr;

	if (cgltf_parse_file(&options, modelPath.c_str(), &data) != cgltf_result_success) {
		throw std::runtime_error("Failed to parse gltf: " + modelPath);
	}

	if (cgltf_load_buffers(&options, data, modelPath.c_str()) != cgltf_result_success) {
		cgltf_free(data);
		throw std::runtime_error("Failed to load gltf buffers: " + modelPath);
	}

	u32 weightBase = 0;
	for (cgltf_size i = 0; i < data->scene->nodes_count; i++) {
		recurseNodes(data->scene->nodes[i], out, weightBase);
	}

	skeleton.data = data;
	skeleton.skin = data->skins_count ? &data->skins[0] : nullptr;
	skeleton.anim = data->animations_count ? &data->animations[0] : nullptr;

	return true;
}

void HTN::Loader::recurseNodes(cgltf_node* node, fModel& out, u32& weightBase) {
	u32 meshTargetCount = 0;

	if (node->mesh) {
		bool isSkinned = (node->skin != nullptr);

		f32 worldCoord[16];
		cgltf_node_transform_world(node, worldCoord);

		enginemath::Mat4 worldTransform(
			enginemath::Vec4(worldCoord[0], worldCoord[1], worldCoord[2], worldCoord[3]),
			enginemath::Vec4(worldCoord[4], worldCoord[5], worldCoord[6], worldCoord[7]),
			enginemath::Vec4(worldCoord[8], worldCoord[9], worldCoord[10], worldCoord[11]),
			enginemath::Vec4(worldCoord[12], worldCoord[13], worldCoord[14], worldCoord[15])
		);

		for (cgltf_size i = 0; i < node->mesh->primitives_count; i++) {
			cgltf_primitive* primitive = &node->mesh->primitives[i];

			cgltf_accessor* posAccessor = nullptr;
			cgltf_accessor* normalAccessor = nullptr;
			cgltf_accessor* texAccessor = nullptr;
			cgltf_accessor* jointAccessor = nullptr;
			cgltf_accessor* weightAccessor = nullptr;

			for (cgltf_size a = 0; a < primitive->attributes_count; a++) {
				cgltf_attribute* attribute = &primitive->attributes[a];
				if (attribute->type == cgltf_attribute_type_position) posAccessor = attribute->data;
				else if (attribute->type == cgltf_attribute_type_normal) normalAccessor = attribute->data;
				else if (attribute->type == cgltf_attribute_type_texcoord) texAccessor = attribute->data;
				else if (attribute->type == cgltf_attribute_type_joints) jointAccessor = attribute->data;
				else if (attribute->type == cgltf_attribute_type_weights) weightAccessor = attribute->data;
			}

			if (!posAccessor) continue;

			u32 indexStart = static_cast<u32>(out.indices.size());
			u32 vertexOffset = static_cast<u32>(out.vertices.size());
			cgltf_size posCount = posAccessor->count;

			for (cgltf_size p = 0; p < posCount; p++) {
				Vertex vertex{};
				cgltf_accessor_read_float(posAccessor, p, vertex.position.elements, 3);
				if (normalAccessor) cgltf_accessor_read_float(normalAccessor, p, vertex.normal.elements, 3);
				if (texAccessor) cgltf_accessor_read_float(texAccessor, p, vertex.texCoord.data, 2);
				if (jointAccessor) cgltf_accessor_read_uint(jointAccessor, p, vertex.joints, 4);
				if (weightAccessor) cgltf_accessor_read_float(weightAccessor, p, vertex.weights.elements, 4);

				if (!isSkinned) {
					enginemath::Vec4 wp = worldTransform * enginemath::Vec4(vertex.position.x, vertex.position.y, vertex.position.z, 1.0f);
					vertex.position = {wp.x, wp.y, wp.z};

					if (normalAccessor) {
						enginemath::Vec4 wn = worldTransform * enginemath::Vec4(vertex.normal.x, vertex.normal.y, vertex.normal.z, 0.0f);
						vertex.normal = {wn.x, wn.y, wn.z};
						vertex.normal.normalize();
					}
				}

				vertex.color = {0.8f, 0.8f, 0.8f};
				out.vertices.push_back(vertex);
			}

			if (primitive->indices) {
				cgltf_size indexCount = primitive->indices->count;
				for (cgltf_size k = 0; k < indexCount; k++) {
					cgltf_size index = cgltf_accessor_read_index(primitive->indices, k);
					out.indices.push_back(static_cast<u32>(index) + vertexOffset);
				}
			} else {
				for (cgltf_size k = 0; k < posCount; k++) {
					out.indices.push_back(static_cast<u32>(k) + vertexOffset);
				}
			}

			u32 indexCount = static_cast<u32>(out.indices.size()) - indexStart;

			cgltf_size targetCount = primitive->targets_count;
			meshTargetCount = static_cast<u32>(targetCount);
			u32 morphStartIndex = targetCount > 0 ? static_cast<u32>(out.deltas.size()) : (u32)-1;

			submesh sm{};
			sm.indexStart = indexStart;
			sm.indexCount = indexCount;
			sm.morphStartIndex = morphStartIndex;
			sm.targetCount = static_cast<u32>(targetCount);
			sm.vertexOffset = vertexOffset;
			sm.vertexCount = static_cast<u32>(posCount);
			sm.weightsStartIndex = weightBase;
			sm.isSkinned = isSkinned ? 1u : 0u;
			out.primitives.push_back(sm);

			for (cgltf_size t = 0; t < targetCount; t++) {
				cgltf_morph_target& target = primitive->targets[t];
				cgltf_accessor* posTargetAccessor = nullptr;

				for (cgltf_size a = 0; a < target.attributes_count; a++) {
					if (target.attributes[a].type == cgltf_attribute_type_position) {
						posTargetAccessor = target.attributes[a].data;
					}
				}

				if (!posTargetAccessor) continue;
				for (cgltf_size d = 0; d < posTargetAccessor->count; d++) {
					enginemath::Vec3 delta;
					cgltf_accessor_read_float(posTargetAccessor, d, delta.elements, 3);

					enginemath::Vec4 wd = isSkinned
						? enginemath::Vec4::toVec4Dir(delta)
						: worldTransform * enginemath::Vec4::toVec4Dir(delta);
					out.deltas.push_back(wd);
				}
			}
		}
	}

	if (node->skin && out.inverseBindMatrix.empty()) {
		cgltf_accessor* ibmAccessor = node->skin->inverse_bind_matrices;
		for (cgltf_size i = 0; i < node->skin->joints_count; i++) {
			f32 ibm[16];
			cgltf_accessor_read_float(ibmAccessor, i, ibm, 16);
			out.inverseBindMatrix.push_back(enginemath::Mat4(
				enginemath::Vec4(ibm[0], ibm[1], ibm[2], ibm[3]),
				enginemath::Vec4(ibm[4], ibm[5], ibm[6], ibm[7]),
				enginemath::Vec4(ibm[8], ibm[9], ibm[10], ibm[11]),
				enginemath::Vec4(ibm[12], ibm[13], ibm[14], ibm[15])
			));
		}
	}

	weightBase += meshTargetCount;

	for (cgltf_size i = 0; i < node->children_count; i++) {
		recurseNodes(node->children[i], out, weightBase);
	}
}

HTN::Skeleton::~Skeleton() {
	if (data) cgltf_free(data);
}

HTN::f32 HTN::Skeleton::animLength() const {
	if (!anim) return 0.0f;
	f32 maxT = 0.0f;
	for (cgltf_size c = 0; c < anim->channels_count; c++) {
		cgltf_accessor* in = anim->channels[c].sampler->input;
		f32 last;
		cgltf_accessor_read_float(in, in->count - 1, &last, 1);
		if (last > maxT) maxT = last;
	}
	return maxT;
}

void HTN::Skeleton::sample(f32 t) {
	if (!anim) return;
	for (cgltf_size c = 0; c < anim->channels_count; c++) {
		cgltf_animation_channel* ch = &anim->channels[c];
		cgltf_animation_sampler* s = ch->sampler;
		cgltf_node* node = ch->target_node;

		int n = (int)s->input->count;
		int i = 0; f32 t0 = 0.0f, t1 = 0.0f;
		for (; i < n - 1; i++) {
			cgltf_accessor_read_float(s->input, i,     &t0, 1);
			cgltf_accessor_read_float(s->input, i + 1, &t1, 1);
			if (t >= t0 && t <= t1) break;
		}
		if (i >= n - 1) i = n - 1;
		int j = (i < n - 1) ? i + 1 : i;
		f32 f = (t1 > t0) ? (t - t0) / (t1 - t0) : 0.0f;

		if (ch->target_path == cgltf_animation_path_type_translation) {
			f32 a[3], b[3];
			cgltf_accessor_read_float(s->output, i, a, 3);
			cgltf_accessor_read_float(s->output, j, b, 3);
			for (int k = 0; k < 3; k++) node->translation[k] = a[k] + (b[k] - a[k]) * f;
		}
		else if (ch->target_path == cgltf_animation_path_type_scale) {
			f32 a[3], b[3];
			cgltf_accessor_read_float(s->output, i, a, 3);
			cgltf_accessor_read_float(s->output, j, b, 3);
			for (int k = 0; k < 3; k++) node->scale[k] = a[k] + (b[k] - a[k]) * f;
		}
		else if (ch->target_path == cgltf_animation_path_type_rotation) {
			f32 a[4], b[4];
			cgltf_accessor_read_float(s->output, i, a, 4);
			cgltf_accessor_read_float(s->output, j, b, 4);
			slerp(node->rotation, a, b, f);
		}
	}
}

std::vector<enginemath::Mat4> HTN::Skeleton::computePalette(f32 elapsedSeconds, const std::vector<enginemath::Mat4>& inverseBind) {
	std::vector<enginemath::Mat4> palette(MAX_JOINTS, enginemath::Mat4::identity());
	if (!skin || !anim) return palette;

	f32 len = animLength();
	f32 t = len > 0.0f ? std::fmod(elapsedSeconds, len) : 0.0f;
	sample(t);

	for (cgltf_size j = 0; j < skin->joints_count && j < MAX_JOINTS && j < inverseBind.size(); j++) {
		f32 world[16];
		cgltf_node_transform_world(skin->joints[j], world);

		enginemath::Mat4 jointWorld(
			enginemath::Vec4(world[0], world[1], world[2], world[3]),
			enginemath::Vec4(world[4], world[5], world[6], world[7]),
			enginemath::Vec4(world[8], world[9], world[10], world[11]),
			enginemath::Vec4(world[12], world[13], world[14], world[15])
		);

		palette[j] = jointWorld * inverseBind[j];
	}
	return palette;
}
