#pragma once
#include <vulkan/vulkan.h>
#include "enginemath/mat4.hpp"

#include "../../defines.hpp"
#include "../device.hpp"
#include "../Helpers/buffers.hpp"

#include <vector>

namespace HTN {
	struct UBO {
		enginemath::Mat4 model = enginemath::Mat4::identity();
		enginemath::Mat4 view = enginemath::Mat4::identity();
		enginemath::Mat4 proj = enginemath::Mat4::identity();
	};

	class Uniform {
	public:
		Uniform(Device& _device);
		~Uniform();
		Uniform(const Uniform&) = delete;
		Uniform& operator=(const Uniform&) = delete;

		void updateUniformBuffer(u32 currentImage, const UBO& ubo);
		std::vector<VkBuffer> getUniformBuffers() { return uniformBuffers; }

	private:
		Device& device;

		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBuffersMemory;
		std::vector<void*> uniformBuffersMapped;

		void createUniformBuffers();
	};
} // namespace HTN
