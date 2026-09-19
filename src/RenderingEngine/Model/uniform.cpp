#include "uniform.hpp"

HTN::Uniform::Uniform(Device& _device) : device(_device) {
	createUniformBuffers();
}

HTN::Uniform::~Uniform() {
	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroyBuffer(device.getDevice(), uniformBuffers[i], nullptr);
		vkFreeMemory(device.getDevice(), uniformBuffersMemory[i], nullptr);
		vkDestroyBuffer(device.getDevice(), lightUniformBuffers[i], nullptr);
		vkFreeMemory(device.getDevice(), lightUniformBuffersMemory[i], nullptr);
	}
}

void HTN::Uniform::createUniformBuffers() {
	VkDeviceSize bufferSize = sizeof(UBO);

	uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
	uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		Buffer::createBuffer(device, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
							 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
							 uniformBuffers[i], uniformBuffersMemory[i]);
		vkMapMemory(device.getDevice(), uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
	}

	VkDeviceSize lightBufferSize = sizeof(LightUBO);

	lightUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
	lightUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
	lightUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		Buffer::createBuffer(device, lightBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
							 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
							 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
							 lightUniformBuffers[i], lightUniformBuffersMemory[i]);
		vkMapMemory(device.getDevice(), lightUniformBuffersMemory[i], 0, lightBufferSize, 0, &lightUniformBuffersMapped[i]);
	}
}

void HTN::Uniform::updateUniformBuffer(u32 currentImage, const UBO& ubo) {
	memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(UBO));
}

void HTN::Uniform::updateLightBuffer(u32 currentImage, const LightUBO& light) {
	memcpy(lightUniformBuffersMapped[currentImage], &light, sizeof(LightUBO));
}
