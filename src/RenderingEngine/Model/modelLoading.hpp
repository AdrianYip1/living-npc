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
		static bool loadModel(const std::string& modelPath, fModel& out);

	private:
		static void recurseNodes(cgltf_node* node, fModel& out, u32& weightBase);
	};
} // namespace HTN
