#pragma once
#include <vulkan/vulkan.h>
#include "enginemath/mat4.hpp"
#include "enginemath/vec3.hpp"

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

	struct LightUBO {
		alignas(16) enginemath::Vec3 position = enginemath::Vec3(0.0f);
		alignas(16) enginemath::Vec3 direction = enginemath::Vec3(1.0f);
		alignas(16) enginemath::Vec3 color = enginemath::Vec3(1.0f);
	};

	class Uniform {
	public:
		Uniform(Device& _device);
		~Uniform();
		Uniform(const Uniform&) = delete;
		Uniform& operator=(const Uniform&) = delete;

		void updateUniformBuffer(u32 currentImage, const UBO& ubo);
		void updateLightBuffer(u32 currentImage, const LightUBO& light);
		std::vector<VkBuffer> getUniformBuffers() { return uniformBuffers; }
		std::vector<VkBuffer> getLightUniformBuffers() { return lightUniformBuffers; }

	private:
		Device& device;

		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBuffersMemory;
		std::vector<void*> uniformBuffersMapped;

		std::vector<VkBuffer> lightUniformBuffers;
		std::vector<VkDeviceMemory> lightUniformBuffersMemory;
		std::vector<void*> lightUniformBuffersMapped;

		void createUniformBuffers();
	};
} // namespace HTN
