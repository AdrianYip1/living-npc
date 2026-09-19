#include "modelLoading.hpp"
#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include <enginemath/vec4.hpp>
#include <enginemath/mat4.hpp>

#include <cstring>
#include <stdexcept>

bool HTN::Loader::loadModel(const std::string& modelPath, fModel& out) {
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

	cgltf_free(data);
	return true;
}

void HTN::Loader::recurseNodes(cgltf_node* node, fModel& out, u32& weightBase) {
	u32 meshTargetCount = 0;

	if (node->mesh) {
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

			for (cgltf_size a = 0; a < primitive->attributes_count; a++) {
				cgltf_attribute* attribute = &primitive->attributes[a];
				if (attribute->type == cgltf_attribute_type_position) posAccessor = attribute->data;
				else if (attribute->type == cgltf_attribute_type_normal) normalAccessor = attribute->data;
				else if (attribute->type == cgltf_attribute_type_texcoord) texAccessor = attribute->data;
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

				enginemath::Vec4 wp = worldTransform * enginemath::Vec4(vertex.position.x, vertex.position.y, vertex.position.z, 1.0f);
				vertex.position = {wp.x, wp.y, wp.z};

				if (normalAccessor) {
					enginemath::Vec4 wn = worldTransform * enginemath::Vec4(vertex.normal.x, vertex.normal.y, vertex.normal.z, 0.0f);
					vertex.normal = {wn.x, wn.y, wn.z};
					vertex.normal.normalize();
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

			out.primitives.push_back({ indexStart, indexCount, morphStartIndex, static_cast<u32>(targetCount),
									   vertexOffset, static_cast<u32>(posCount), weightBase });

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

					enginemath::Vec4 wd = worldTransform * enginemath::Vec4::toVec4Dir(delta);
					out.deltas.push_back(wd);
				}
			}
		}
	}

	weightBase += meshTargetCount;

	for (cgltf_size i = 0; i < node->children_count; i++) {
		recurseNodes(node->children[i], out, weightBase);
	}
}
