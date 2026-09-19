#pragma once
#include <vulkan/vulkan.h>

#include "../../defines.hpp"
#include "model.hpp"

#include <string>
#include <vector>

struct cgltf_node;
struct cgltf_data;

namespace HTN {
	class Loader {
	public:
		static bool loadModel(const std::string& modelPath,
							  std::vector<Vertex>& vertices,
							  std::vector<u32>& indices);

	private:
		static void recurseNodes(cgltf_node* node,
								 std::vector<Vertex>& vertices,
								 std::vector<u32>& indices);
	};
} // namespace HTN
