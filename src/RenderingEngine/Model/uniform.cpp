#include "uniform.hpp"
#include "../../npcConfig.hpp"

HTN::Uniform::Uniform(Device& _device) : device(_device) {
	createUniformBuffers();
}

HTN::Uniform::~Uniform() {
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroyBuffer(device.getDevice(), uniformBuffers[i], nullptr);
		vkFreeMemory(device.getDevice(), uniformBuffersMemory[i], nullptr);
		vkDestroyBuffer(device.getDevice(), lightUniformBuffers[i], nullptr);
		vkFreeMemory(device.getDevice(), lightUniformBuffersMemory[i], nullptr);
		vkDestroyBuffer(device.getDevice(), weightStorageBuffers[i], nullptr);
		vkFreeMemory(device.getDevice(), weightStorageBuffersMemory[i], nullptr);
		vkDestroyBuffer(device.getDevice(), jointStorageBuffers[i], nullptr);
		vkFreeMemory(device.getDevice(), jointStorageBuffersMemory[i], nullptr);
	}
}

void HTN::Uniform::createUniformBuffers() {
	VkDeviceSize bufferSize = sizeof(UBO);
	VkDeviceSize lightBufferSize = sizeof(LightUBO);
	VkDeviceSize weightBufferSize = sizeof(f32) * MAX_WEIGHTS * npcCount();
	VkDeviceSize jointBufferSize = sizeof(enginemath::Mat4) * MAX_JOINTS * npcCount();

	uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
	uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

	lightUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	lightUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
	lightUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

	weightStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	weightStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
	weightStorageBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

	jointStorageBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	jointStorageBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
	jointStorageBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		Buffer::createBuffer(device, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
							 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
							 uniformBuffers[i], uniformBuffersMemory[i]);
		vkMapMemory(device.getDevice(), uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);

		Buffer::createBuffer(device, lightBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
							 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
							 lightUniformBuffers[i], lightUniformBuffersMemory[i]);
		vkMapMemory(device.getDevice(), lightUniformBuffersMemory[i], 0, lightBufferSize, 0, &lightUniformBuffersMapped[i]);

		Buffer::createBuffer(device, weightBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
							 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
							 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
							 weightStorageBuffers[i], weightStorageBuffersMemory[i]);
		vkMapMemory(device.getDevice(), weightStorageBuffersMemory[i], 0, weightBufferSize, 0, &weightStorageBuffersMapped[i]);
		memset(weightStorageBuffersMapped[i], 0, weightBufferSize);

		Buffer::createBuffer(device, jointBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
							 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
							 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
							 jointStorageBuffers[i], jointStorageBuffersMemory[i]);
		vkMapMemory(device.getDevice(), jointStorageBuffersMemory[i], 0, jointBufferSize, 0, &jointStorageBuffersMapped[i]);
	}
}

void HTN::Uniform::updateUniformBuffer(u32 currentImage, const UBO& ubo) {
	memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(UBO));
}

void HTN::Uniform::updateLightBuffer(u32 currentImage, const LightUBO& light) {
	memcpy(lightUniformBuffersMapped[currentImage], &light, sizeof(LightUBO));
}

void HTN::Uniform::updateWeightBuffer(u32 currentImage, u32 headIndex, const std::vector<f32>& weights) {
	char* dst = static_cast<char*>(weightStorageBuffersMapped[currentImage])
		+ headIndex * MAX_WEIGHTS * sizeof(f32);
	memcpy(dst, weights.data(), sizeof(f32) * weights.size());
}

void HTN::Uniform::updateJointBuffer(u32 currentImage, const std::vector<enginemath::Mat4>& palette) {
	memcpy(jointStorageBuffersMapped[currentImage], palette.data(), sizeof(enginemath::Mat4) * palette.size());
}
