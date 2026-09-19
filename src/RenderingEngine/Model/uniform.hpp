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
		f32 time = 0.0f;
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
		void updateWeightBuffer(u32 currentImage, const std::vector<f32>& weights);
		void updateJointBuffer(u32 currentImage, const std::vector<enginemath::Mat4>& palette);
		std::vector<VkBuffer> getUniformBuffers() { return uniformBuffers; }
		std::vector<VkBuffer> getLightUniformBuffers() { return lightUniformBuffers; }
		std::vector<VkBuffer> getWeightBuffers() { return weightStorageBuffers; }
		std::vector<VkBuffer> getJointBuffers() { return jointStorageBuffers; }

	private:
		Device& device;

		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBuffersMemory;
		std::vector<void*> uniformBuffersMapped;

		std::vector<VkBuffer> lightUniformBuffers;
		std::vector<VkDeviceMemory> lightUniformBuffersMemory;
		std::vector<void*> lightUniformBuffersMapped;

		std::vector<VkBuffer> weightStorageBuffers;
		std::vector<VkDeviceMemory> weightStorageBuffersMemory;
		std::vector<void*> weightStorageBuffersMapped;

		std::vector<VkBuffer> jointStorageBuffers;
		std::vector<VkDeviceMemory> jointStorageBuffersMemory;
		std::vector<void*> jointStorageBuffersMapped;

		void createUniformBuffers();
	};
} // namespace HTN
