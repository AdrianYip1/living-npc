#pragma once
#include <vulkan/vulkan.h>
#include <enginemath/vec4.hpp>
#include <enginemath/mat4.hpp>

#include "../../defines.hpp"
#include "model.hpp"

#include <string>
#include <vector>
#include <unordered_map>

struct cgltf_node;
struct cgltf_mesh;
struct cgltf_data;
struct cgltf_skin;
struct cgltf_animation;

namespace HTN {

	class Skeleton {
	public:
		Skeleton() = default;
		~Skeleton();
		Skeleton(const Skeleton&) = delete;
		Skeleton& operator=(const Skeleton&) = delete;

		f32 animLength() const;
		void sample(f32 t);
		std::vector<enginemath::Mat4> computePalette(f32 elapsedSeconds, const std::vector<enginemath::Mat4>& inverseBind);

		cgltf_data* data = nullptr;
		cgltf_skin* skin = nullptr;
		cgltf_animation* anim = nullptr;
	};

	class Loader {
	public:
		static bool loadModel(const std::string& modelPath, fModel& out, Skeleton& skeleton, bool instanced = false);

	private:
		static void recurseNodes(cgltf_node* node, fModel& out, u32& weightBase);
		static void collectMeshInstances(cgltf_node* node, std::vector<cgltf_mesh*>& order,
			std::unordered_map<cgltf_mesh*, std::vector<enginemath::Mat4>>& instances);
		static void buildInstancedMesh(cgltf_mesh* mesh, fModel& out,
			const std::vector<enginemath::Mat4>& transforms);
		static void collectCollision(cgltf_node* node, fModel& out);
	};
} // namespace HTN
